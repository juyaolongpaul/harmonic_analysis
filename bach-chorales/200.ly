\version "2.10.33"

\header {
  title = "200 - Christus ist erstanden, hat überwunden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  c4 g4 a4 b4 
  c2 c4 g4 
  c4 bes4 a2 
  g2 g4 a4 
  bes4 c4 bes4 a4   %5
  g2 g4 c4 
  c4 b4 c2 
  c4 c4 d4 d4 
  c4 bes4 a2 
  g4 a4 bes4 a4   %10
  g2 bes4 c4 
  d4 ees8 d8 c2 
  d2 d4 c4 
  d4 c4 bes4 g4 
  a4 b4 c2   %15
  
}

alto = \relative c'' {
  g4 g4. fis8 g4 
  g4. f4 e8 e4~ 
  e8 fis8 g4 g4 fis4 
  d2 d8 ees4 d8~ 
  d8 bes'8 a8 g8 fis8 g4 fis8   %5
  d2 e8 f8 g4 
  g4 g4 g2 
  a4 a8 g8 f8 g8 a4~ 
  a4 bes8 g8 f2 
  e4 e8 fis8 g4. fis8   %10
  g2 d8 bes'4 a8 
  bes2 bes4 a4 
  bes2 bes4. a8 
  bes4 a4 g4 e4 
  f8 e8 d8 f8 e2   %15
  
}

tenor = \relative c' {
  e4 d4 c4 d4 
  c8 bes8 a4 g4 c4 
  c4 d4 ees4 d8 c8 
  bes2 bes4 c4 
  bes8 g8 ees'4 d4~ d8. c16   %5
  bes2 c4. d8 
  e8 d16 c16 d8 f4 e16 d16 e4 
  f4 f4 f8 e8 d8 e8 
  f4. e8 c2 
  c4 c4 d4 ees8 d16 c16   %10
  bes2 bes8 d8 f4 
  f4 ees8 f8 g8 ees8 c8 f8 
  f2 f4 f4 
  f4 ees8 d8 d4 c4 
  c4 g4 g2   %15
  
}

baixo = \relative c' {
  c4 b4 a4 g8 f8 
  e4 f4 c4 c'8 bes8 
  a4 g4 c,4 d4 
  g,2 g'4 fis4 
  g4 c,4 d4 d,4   %5
  g2 c8 d8 e8 f8 
  g4 g,4 c2 
  f8 g8 f8 e8 d8 e8 f8 g8 
  a8 f8 g8 c,8 f2 
  c'8 bes8 a4 g8 bes,8 c8 d8   %10
  ees2 g4 f4 
  bes8 a8 g8 f8 ees8 c8 f4 
  bes,2 bes'4 f4 
  bes8 a8 g8 fis8 g8 a8 bes8 c8 
  f,4 g4 c,2   %15
  
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}
\version "2.10.33"

\header {
  title = "176 - Erstanden ist der heil'ge Christ"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key f \major
}


soprano = \relative c' {
  \partial 4 f4 
    f4. g8 a8 bes8 
  c2 g4 
  a2 b4 
  c2 bes4 
  a4. g8 a8 bes8   %5
  c2 bes4 
  a2 g4 
  a2 a4 
  c2 bes4 
  a2 g4   %10
  f2 e4 
  f2 f4 
  g8 f8 g8 a8 bes4 
  a2 g4 
  f2 e4   %15
  f2 
  
}

alto = \relative c' {
  \partial 4 c4 
    c4 d8 e8 f4~ 
  f4 e8 f8 g4~ 
  g8 e8 f2~ 
  f4 e4 g4~ 
  g4 f2   %5
  g8 f8 g8 a4 g8~ 
  g8 f16 e16 f4. e16 d16 
  cis2 d4 
  e4. f8 g4~ 
  g4 f4 e4~   %10
  e4 d4 c8 bes8 
  a2 d4 
  d4 e8 f8 d8 e8 
  f2 e4~ 
  e4 d4 c4   %15
  c2 
  
}

tenor = \relative c' {
  \partial 4 a4 
    a8 bes8 a8 g8 f4 
  g2 c4 
  c2 d4 
  g,2 c4 
  c8 a8 d2   %5
  c8 d8 e4 d4 
  d4 a4 d,4 
  e2 f4 
  a4 c2~ 
  c8 a8 d8 c8 bes4   %10
  a8 f8 bes8 a8 g4 
  f2 a4 
  g4 c4 d8 g,8 
  a8 bes8 c4. bes8 
  a4 bes8 a8 g8 bes8   %15
  a2 
  
}

baixo = \relative c, {
  \partial 4 f4 
    f'4. e8 d4 
  c4. d8 e8 c8 
  f4. e8 d4 
  c2 e4 
  f4. e8 d4   %5
  e4. fis8 g4 
  d4. c8 bes4 
  a2 d4 
  a'8 a,8 a'8 g8 f8 e8 
  f8 f,8 f'8 e8 d8 cis8   %10
  d4 bes4 c4 
  d2 d8 c8 
  bes4. a8 g4 
  f8 g8 a8 bes8 c4 
  d8 c8 bes4 c4   %15
  f,2 
  
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
\version "2.10.33"

\header {
  title = "173 - O Herzensangst"
  composer = "J. S. Bach"
}


global = {
  \time 3/4
  \key ees \major
}


soprano = \relative c'' {
  bes4 g4 c4 
  bes2 ees,4 
  f2 g4 
  aes2 bes4 
  g4 ees2   %5
  f4 g4 a4 
  bes2 d4 
  c4 d4 ees4 
  d4 c2 
  bes2.   %10
  bes4 b4 b4 
  c2 d4 
  ees4. ees8 f4 
  ees4 d2 
  c2 bes4   %15
  c4. d8 ees4 
  ees2 d4 
  ees2. 
  
}

alto = \relative c' {
  ees4 ees4 ees4 
  d2 ees4 
  d8 c8 bes2 
  f'2 f4 
  ees4 bes2   %5
  d4 ees4 ees4 
  d2 f4 
  f4 f4 ees4 
  f4 g4 f8 ees8 
  d2.   %10
  f4 g4 g8 f8 
  ees2 g4 
  g4 g4 f4 
  g4 aes4 g8 f8 
  ees2 ees4   %15
  ees4 f4 g4~ 
  g4 f8 ees8 f8 aes8 
  g2. 
  
}

tenor = \relative c' {
  g4 bes4 aes8 g8 
  f2 bes4 
  aes2 g4 
  c4 bes4 bes4 
  bes4 g2   %5
  bes4 bes4 c4 
  f,2 bes4 
  c4 bes4 bes4 
  bes2 a4 
  f2.   %10
  d'4 d4 g,4 
  g2 d'4 
  c4 c4 c4 
  c2 b4 
  g2 des'4   %15
  c8 bes8 c4 bes4 
  bes2 bes4 
  bes2. 
  
}

baixo = \relative c {
  ees4 ees4 aes,4 
  aes2 g'4 
  f8 ees8 d4 ees4~ 
  ees4 d8 c8 d4 
  ees4 ees,2   %5
  bes'4 ees8 d8 c4 
  bes2 bes'4 
  a4 aes4 g4 
  f4 ees4 f4 
  bes,2.   %10
  bes'8 aes8 g8 f8 ees8 d8 
  c2 b4 
  c4. bes8 aes4 
  g4 f4 g4 
  c2 g4   %15
  aes2 g8 aes8 
  bes2. 
  ees,2. 
  
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
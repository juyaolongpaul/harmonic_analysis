\version "2.10.33"

\header {
  title = "118 - In dich hab ich gehoffet, Herr 1"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key bes \major
}


soprano = \relative c'' {
  \partial 4 bes4 
    bes4 f'4 f4 ees8 d8 
  c4 d4 ees8 d8 c4 
  bes4 bes4 c4 d4 
  ees4 c4 f4 g4 
  f4 d4 d8 ees8 f4   %5
  ees4 d8 c8 bes8 c8 d8 c8 
  c4 d4 bes8 c8 d4 
  ees4 d4 c4 d4 
  ees4 d4 c4 bes4 
  f'4. ees8 d4 ees8 d8   %10
  c2 bes2 
  
}

alto = \relative c' {
  \partial 4 f4 
    g4 a8 bes8 c4 bes8 a8 
  g4 f4 bes4 a4 
  f4 g4 fis4 f4 
  ees8 f8 g4 f4. e8 
  a4 a4 g4 f4   %5
  c'4 bes4 d,4 g4 
  f4 f4 g8 a8 b4 
  c4 bes4 bes4 aes4 
  g4 f4 f8 ees8 d8 ees8 
  f8 g8 a2 g4   %10
  g4 f8 ees8 d2 
  
}

tenor = \relative c' {
  \partial 4 d4 
    d8 ees8 f4 f4 g8 f8 
  ees4 f4 bes,4 f'8 ees8 
  d4 ees4 ees4 d4 
  g,4 g8 a8 bes8 c8 d8 c8 
  c4 c4 bes8 c8 d8 ees8   %5
  f4 f4 g4 bes,4 
  a4 bes8 a8 g4 f4 
  g4 g'8 f8 ees4 f4 
  bes,4 bes4 a4 bes4 
  c2. bes4   %10
  bes4 a4 f2 
  
}

baixo = \relative c' {
  \partial 4 bes8  a8 
    g4 d4 a'4 bes4~ 
  bes4 aes4 g8 ees8 f4 
  bes,4 ees4 a,4 b4 
  c8 d8 ees4 d8 c8 bes8 c8 
  f4 fis4 g4 aes4   %5
  a4 bes8 a8 g8 f8 e4 
  f4 bes,4 ees4 d4 
  c4 g'4 aes8 g8 f4 
  ees4 bes4 f'4 g4 
  a8 g8 f4 fis4 g8 f8   %10
  e4 f4 bes,2 
  
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
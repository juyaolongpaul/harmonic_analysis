\version "2.10.33"

\header {
  title = "281 - Wo soll ich fliehen hin"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  \partial 4 g4 
  g4 a4 bes4 c4 
  d2. d4 
  d4 c4 bes4 c4 
  ais2. ais4 
  bes4 c4 d4 d4   %5
  c2 d4 d4 
  bes4 c4 d4 d4 
  c2 bes4 d4 
  f4 d4 d4 d4 
  c2 c4 c4   %10
  d4 c4 bes4 c8 bes8 
  a2 g4 

}

alto = \relative c' {
  \partial 4 d4 
  d4 d4 d4 g4 
  fis2. fis4 
  g4 g8 fis8 g4 a4 
  f2. f4 
  g4 a4 bes4 bes4   %5
  bes4 a4 bes4 a4 
  g4 a8 g8 f4 f4 
  g4 f4 f4 bes4 
  c4 bes4 a4 g4 
  g4 f8 e8 f4 g4   %10
  fis4 fis4 g4 g4 
  g4 fis4 d4 

}

tenor = \relative c' {
  \partial 4 bes4 
  bes4 a4 g8 bes8 ees4 
  a,2. a4 
  bes4 c4 d4 ees4 
  d2. d4 
  d4 f4 f4 g4   %5
  g4 f4 f4 d4 
  d4 f4 bes,4 bes4 
  bes4 a4 d4 f4 
  f4 f8 g8 a8 a,8 bes4 
  bes4 a8 g8 a4 g4   %10
  a4 d4 d4 c8 d8 
  ees4 d8 c8 b4 

}

baixo = \relative c {
  \partial 4 g4 
  g'4 fis4 g4 ees4 
  d2. c4 
  bes4 a4 g4 c4 
  d2. d4 
  g4 f4 bes8 a8 g8 f8   %5
  ees4 f4 bes,4 fis'4 
  g4 f8 ees8 d8 c8 bes8 d8 
  ees8 c8 f4 bes,4 bes'4 
  a4 bes4 fis4 g4 
  e4 c4 f4 ees4   %10
  d8 c8 bes8 a8 g8 f'8 ees8 d8 
  c4 d4 g,4 

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